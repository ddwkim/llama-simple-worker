#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "arg.h"
#include "common.h"
#include "llama-vocab.h" // From third-party/llama.cpp/src
#include "llama.h"       // From third-party/llama.cpp/include
#include "log.h"         // From third-party/llama.cpp/common
#include "sampling.h"


#define INF 1e9
#define NUM_THREADS 2

using job_idx_t = int;

std::string test_system_prompt  = "The following is an article about roofline model.";


std::string prompt = R"(
The Roofline model is a visual model used to analyze and optimize the performance of 
computational kernels on various hardware platforms.)";

template <typename T> class thread_safe_queue
{
  public:
    void push(const T &item)
    {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            queue_.push(item);
        }
        cv_.notify_one();
    }

    std::optional<T> pop()
    {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this]() { return !queue_.empty() || !running_; });

        if (queue_.empty())
            return std::nullopt;

        T item = queue_.front();
        queue_.pop();

        return item;
    }

    void stop()
    {
        std::unique_lock<std::mutex> lock(mtx_);
        running_ = false;
        cv_.notify_all();
    }

  private:
    std::queue<T> queue_;
    std::mutex mtx_;
    std::condition_variable cv_;
    bool running_ = true;
};

llama_chat_message generate_chat_message(const std::string &base_prompt, llama_model *model, const std::string &role)
{
    llama_chat_message msg;
    msg.role = role.c_str();
    msg.content = base_prompt.c_str();
    return msg;
}

std::vector<char> generate_message_and_apply_template(const std::string &base_prompt, llama_model *model,
                                                      size_t initial_buf_size, const std::string &role)
{
    llama_chat_message msg = generate_chat_message(base_prompt, model, role);

    std::vector<char> buf(initial_buf_size);
    int str_len = 0;

    while (true)
    {
        str_len = llama_chat_apply_template(model, nullptr, &msg, 1, false, buf.data(), buf.size());
        if (str_len < 0)
        {
            throw std::runtime_error("llama_chat_apply_template failed");
        }
        if (static_cast<size_t>(str_len) < buf.size())
        {
            break;
        }
        buf.resize(buf.size() * 2);
    }
    
    buf[str_len] = '\0';

    buf.resize(str_len + 1);

    return buf;
}

class llama_worker
{
  private:
    // **stateful**
    common_params params_;
    llama_model *model_;
    llama_context *ctx_;
    int n_ctx_;

    struct common_sampler *smpl_;

    llama_batch batch_;
    common_init_result llama_init_;

    // **Non-stateful**
    std::vector<llama_token> tokens_system_;
    int n_tokens_system_;

    std::string response_ = "";

    bool is_finished_ = false;
    std::string error_message_;

  public:
    llama_worker(common_params params)
        : params_(params)    {

        LOG_INF("Begin Initialize llama worker\n");

        // params_.warmup = false;
        llama_init_ = common_init_from_params(params_);

        model_ = llama_init_.model;
        ctx_ = llama_init_.context;
        n_ctx_ = llama_n_ctx(ctx_);

        smpl_ = common_sampler_init(model_, params_.sparams);

        batch_ = llama_batch_init(n_ctx_, 0, 1);

        llama_synchronize(ctx_);

        LOG_INF("Initialized llama worker\n");
    }

    ~llama_worker()
    {
        llama_batch_free(batch_);

        llama_free(ctx_);
        llama_free_model(model_);
    }

    bool initialize()
    {
        tokens_system_ = common_tokenize(
            ctx_, generate_message_and_apply_template(test_system_prompt, model_, 1024, "system").data(), false,
            true);
        n_tokens_system_ = tokens_system_.size();

        for (int32_t i = 0; i < n_tokens_system_; ++i)
        {
            common_batch_add(batch_, tokens_system_[i], i, {0}, false);
        }

        if (llama_decode(ctx_, batch_))
        {
            error_message_ = std::string(__func__) + " : failed to eval, return code " + std::to_string(1);
            return false;
        }

        common_batch_clear(batch_);

        common_sampler_reset(smpl_);

        return true;
    }

    bool infer(std::string prompt)
    {
        auto parse_st = std::chrono::high_resolution_clock::now();

        std::vector<char> tmp = generate_message_and_apply_template(prompt, model_, 1024, "user");

        std::string tmp_str(tmp.data());

        prompt = tmp_str + "<|start_header_id|>assistant<|end_header_id|>\n\n";

        std::cout << prompt << std::endl;

        std::vector<llama_token> prompt_tokens = common_tokenize(ctx_, prompt, false, true);
        const int32_t n_prompt = prompt_tokens.size();

        int total_tokens = n_tokens_system_ + n_prompt;

        if (total_tokens > params_.n_batch || total_tokens > n_ctx_)
        {
            error_message_ = "Number of tokens in batch exceeds the context size, random sample";
            return false;
        }

        LOG_INF("Parsed input in %ld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(
                                                std::chrono::high_resolution_clock::now() - parse_st)
                                                .count());

        for (size_t i = 0; i < n_prompt; ++i)
        {
            common_batch_add(batch_, prompt_tokens[i], i + n_tokens_system_, {0}, false);
        }

        int n_pos = n_tokens_system_;

        while (!this->is_finished_)
        {
            auto decode_st = std::chrono::high_resolution_clock::now();


            // extract the logits only for the last token
            if (batch_.n_tokens > 0)
            {
                batch_.logits[batch_.n_tokens - 1] = true;
            }

            if (!this->process_batch(n_pos))
            {
                // inference failed
                common_sampler_reset(smpl_);
                llama_kv_cache_seq_rm(ctx_, 0, n_tokens_system_, -1);

                return false;
            }

            LOG_INF("Decoded in %ld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(
                                               std::chrono::high_resolution_clock::now() - decode_st)
                                               .count());
            is_finished_ = true;
        }

        llama_kv_cache_seq_rm(ctx_, 0, n_tokens_system_, -1);

        return true;
    }

    bool refresh()
    {
        common_sampler_reset(smpl_);
        common_batch_clear(batch_);

        is_finished_ = false;
        response_ = "";

        return true;
    }

    std::string get_response() const
    {
        return response_;
    }

    std::string get_error_message() const
    {
        return error_message_;
    }


    bool process_batch(int n_pos)
    {
        llama_token new_token_id;
        int n_decode = 0;
        int total_tokens = batch_.n_tokens;
        std::string token_str;

        while (n_pos + batch_.n_tokens < total_tokens + params_.n_predict)
        {
            if (llama_decode(ctx_, batch_))
            {

                error_message_ = std::string(__func__) + " : failed to eval, return code " + std::to_string(1);

                return false;
            }

            n_pos += batch_.n_tokens;

            {
                new_token_id = common_sampler_sample(smpl_, ctx_, -1);

                common_sampler_accept(smpl_, new_token_id, false);

                if (llama_token_is_eog(model_, new_token_id))
                {
                    break;
                }
                token_str = common_token_to_piece(ctx_, new_token_id);
                response_ += token_str;
            }

            common_batch_clear(batch_);

            common_batch_add(batch_, new_token_id, n_pos, {0}, true);
        }

        common_batch_clear(batch_);

        return true;
    }

    bool postprocess()
    {
        return true;
    }
};

void worker_thread_func(int worker_id, llama_worker &worker, thread_safe_queue<std::pair<job_idx_t, std::string>> &job_queue,
                        thread_safe_queue<std::pair<job_idx_t, std::string>> &result_queue)
{
    while (true)
    {
        auto job_opt = job_queue.pop();
        if (!job_opt.has_value())
        {
            LOG_INF("Worker %d exiting.\n", worker_id);

            break;
        }

        std::string job = job_opt.value().second;

        LOG_INF("Inference of job %d by worker  %d \n", job_opt.value().first, worker_id);

        if (!worker.infer(job))
        {
            LOG_INF("Worker %d failed to process job: %s \n", worker_id, worker.get_error_message().c_str());
        }
        else
        {
            result_queue.push(std::make_pair(job_opt.value().first, worker.get_response()));

            LOG_INF("Worker %d processed job successfully.\n", worker_id);
        }

        worker.refresh();
    }

    return;
}

int main(int argc, char **argv)
{
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PARALLEL, NULL))
    {
        return 1;
    }

    // setting log level
    common_init();

    llama_backend_init();

    llama_numa_init(params.numa);

    const int32_t n_seq = params.n_sequences;

    int cnt = 0;

    // Set up thread-safe job queue
    thread_safe_queue<std::pair<job_idx_t, std::string>> job_queue;
    thread_safe_queue<std::pair<job_idx_t, std::string>> result_queue;

    std::vector<std::thread> threads;
    std::vector<std::unique_ptr<llama_worker>> workers;

    for (int i = 0; i < NUM_THREADS; ++i)
    {
        try
        {
            workers.emplace_back(
                std::make_unique<llama_worker>(params));

            if (!workers.back()->initialize())
            {
                std::cerr << "Failed to initialize worker " << i << ": " << workers.back()->get_error_message() << "\n";
                return -1;
            }

            threads.emplace_back(worker_thread_func, i, std::ref(*workers.back()), std::ref(job_queue),
                                 std::ref(result_queue));
        }
        catch (const std::exception &e)
        {
            std::cerr << "Exception during worker " << i << " initialization: " << e.what() << "\n";
            return -1;
        }
    }

    LOG_INF("Workers initialized\n");

    for (int i = 0; i < n_seq; ++i)
    {
        job_queue.push(std::make_pair(cnt++, prompt));
    }

    LOG_INF("Jobs pushed to queue\n");

    job_queue.stop();

    auto st = std::chrono::high_resolution_clock::now();

    for (auto &thread : threads)
    {
        thread.join();
    }

    LOG_INF(
        "All threads finished in %ld ms procssed %d\n",
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - st).count(),
        n_seq);

    result_queue.stop();

    for (int i = 0; i < n_seq; ++i)
    {
        auto result = result_queue.pop();
        if (result.has_value())
        {
            LOG_INF("Job %d processed successfully, response: %s\n", result.value().first, result.value().second.c_str());
        }
    }

    llama_backend_free();

    return 0;
};