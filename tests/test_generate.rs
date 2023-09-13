use llama_cpp_sys::*;
use rand::Rng;
use std::ffi::{c_char, CStr, CString};
use std::process::exit;
use std::env;

/// Based on: https://github.com/ggerganov/llama.cpp/blob/master/examples/simple/simple.cpp
#[test]
pub fn main() {
    unsafe {
        llama_backend_init(true);

        let mut ctx_params = llama_context_default_params();
        ctx_params.n_batch = 512;
        ctx_params.n_gpu_layers = 32;

        let model_path = "models/model.gguf";
        let model_path_cstr = make_c_str(model_path);
        let model = llama_load_model_from_file(model_path_cstr.as_ptr(), ctx_params);

        if model.is_null() {
            println!("error: failed to load model");
            exit(1);
        }

        let prompt_str = make_c_str("### Instruction:\nImplement the following code in typescript: ```### Instruction:\nThe following high level steps are required to implement a raytracing engine:\n### Response:");
        let ctx = llama_new_context_with_model(model, ctx_params);
        let mut tokens_list: Vec<llama_token> = vec![0; 2048];
        let mut used_length = llama_tokenize(ctx, prompt_str.as_ptr(), tokens_list.as_mut_ptr(), tokens_list.len() as i32, true) as usize;

        let max_context_size = llama_n_ctx(ctx);
        let max_tokens_list_size = (max_context_size - 4).max(0) as usize;

        if used_length > max_tokens_list_size {
            println!("error: prompt too long ({} tokens, max {})", used_length, max_tokens_list_size);
            exit(1);
        }

        let mut token_value_buffer: Vec<c_char> = vec![0; 2048];
        let used_tokens = &tokens_list[0..used_length];
        for token in used_tokens.iter() {
            print_c_str(*token, &mut token_value_buffer, ctx);
        }

        // main loop

        // The LLM keeps a contextual cache memory of previous token evaluation.
        // Usually, once this cache is full, it is required to recompute a compressed context based on previous
        // tokens (see "infinite text generation via context swapping" in the main example), but in this minimalist
        // example, we will just stop the loop once this cache is full or once an end of stream is detected.
        let n_gen = max_context_size.max(32);
        println!("\n ... generating {} tokens...\n", n_gen);

        let n_vocab = llama_n_vocab(ctx);
        let mut candidates: Vec<llama_token_data> = Vec::with_capacity(n_vocab as usize);

        let token_history_max_length = 128;
        let mut token_history: Vec<llama_token> = Vec::new();

        while llama_get_kv_cache_token_count(ctx) < n_gen {
            // Use all generated tokens as context for next token
            let existing_token_count = llama_get_kv_cache_token_count(ctx);

            let used_tokens = &tokens_list[0..used_length];
            let result_code = llama_eval(ctx, used_tokens.as_ptr(), used_length as i32, existing_token_count, 8);
            if result_code == 1 {
                println!("eval failed");
                exit(1);
            }

            tokens_list.clear();
            used_length = 0;

            // sample the next token
            let mut new_token_id: llama_token = 0;
            let logits = llama_get_logits(ctx);

            candidates.clear();
            for token_id in 0..n_vocab {
                candidates.push(llama_token_data {
                    id: token_id,
                    logit: *logits.offset(token_id as isize),
                    p: 0f32,
                });
            }

            let mut candidates_p = llama_token_data_array {
                data: candidates.as_mut_ptr(),
                size: candidates.len(),
                sorted: false,
            };

            let repeat_penalty_scan_length = token_history_max_length.min(token_history.len());
            let repeat_penalty_factor = 1.1f32;
            llama_sample_repetition_penalty(
                ctx,
                &mut candidates_p,
                token_history.as_ptr(),
                repeat_penalty_scan_length, 
                repeat_penalty_factor
            );

            new_token_id = llama_sample_token_greedy(ctx, &mut candidates_p);

            // is it an end of stream ?
            if new_token_id == llama_token_eos(ctx) {
                println!("[end of text]");
                break;
            }

            // print the new token :
            print_c_str(new_token_id, &mut token_value_buffer, ctx);

            // push this new token for next evaluation
            tokens_list.push(new_token_id);
            used_length += 1;

            token_history.push(new_token_id);
            if token_history.len() > token_history_max_length {
                token_history.remove(0);
            }
        }

        llama_free(ctx);
        llama_free_model(model);

        llama_backend_free();
    }
}

fn make_c_str(value: &str) -> CString {
    CString::new(value).unwrap()
}

unsafe fn print_c_str(token_id: llama_token, buffer: &mut Vec<c_char>, ctx: *const llama_context) {
    buffer.fill(0);
    llama_token_to_piece(ctx, token_id, buffer.as_mut_ptr(), buffer.len() as i32);
    let c_str = unsafe { CStr::from_ptr(buffer.as_ptr()) };
    let str_slice: &str = c_str.to_str().unwrap();
    print!("{}", str_slice);
}
