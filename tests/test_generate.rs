use core::slice;
use llama_cpp_sys::*;
use std::{
    ffi::{CStr, CString},
    io::{stdout, Write},
};

const BATCH_TOKENS: usize = 512;
const N_LEN: i32 = 32;

/// Adapted from: https://github.com/ggerganov/llama.cpp/blob/master/examples/simple/simple.cpp
#[test]
pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Prompt

    const PROMPT:&str = "Do you like green eggs and ham?";

    // Global init

    unsafe {
        llama_backend_init();
        llama_numa_init(ggml_numa_strategy_GGML_NUMA_STRATEGY_DISABLED);
    }

    // Init model

    let model_params = unsafe { llama_model_default_params() };
    let model_name = CString::new("models/model.gguf").unwrap();
    let model = unsafe { llama_load_model_from_file(model_name.as_ptr(), model_params) };
    if model.is_null() {
        return Err(format!("error: unable to load model: {}", model_name.to_str().unwrap()).into());
    }

    // Init context

    let mut ctx_params = unsafe { llama_context_default_params() };

    ctx_params.seed = 1234;
    ctx_params.n_ctx = 2048;
    ctx_params.n_threads = 8;
    ctx_params.n_threads_batch = 8;

    let ctx = unsafe { llama_new_context_with_model(model, ctx_params) };
    if ctx.is_null() {
        return Err("error: unable to create model context".into());
    }

    // Tokenize the prompt

    let mut tokens_list = tokenize(model, PROMPT, true);

    // Make sure the KV cache is big enough to hold the prompt and generated tokens

    let n_ctx = unsafe { llama_n_ctx(ctx) };
    let n_kv_req = tokens_list.len() + (N_LEN as usize - tokens_list.len());

    println!("n_len = {N_LEN}, n_ctx = {n_ctx}, n_kv_req = {n_kv_req}");

    // Make sure the KV cache is big enough to hold all the prompt and generated tokens

    if n_kv_req > n_ctx.try_into().unwrap() {
        return Err(format!("error: n_kv_req > n_ctx, the required KV cache size is not big enough. Either reduce n_len or increase n_ctx.").into());
    }

    // Print the prompt token-by-token.

    print!("\n");

    for &token in &tokens_list {
        print!("{}", token_to_piece(token, model));
    }

    stdout().flush().unwrap();

    // Create a llama_batch with size 512. We use this object to submit
    // token data for decoding.
    let mut batch = unsafe { llama_batch_init(BATCH_TOKENS as i32, 0, 1) };

    // Evaluate the initial prompt
    for (pos, &token) in tokens_list.iter().enumerate() {
        llama_batch_add(&mut batch, token, pos, false);
    }

    // llama_decode will output logits only for the last token of the prompt
    // (add the last token with logits = true)
    {
        let logits = unsafe {
            slice::from_raw_parts_mut(batch.logits, BATCH_TOKENS)
        };

        logits[usize::try_from(batch.n_tokens - 1).unwrap()] = 1;
    }

    if unsafe { llama_decode(ctx, batch) } != 0 {
        return Err("llama_decode failed".try_into().unwrap());
    }

    // Main loop
    let n_vocab = unsafe { llama_n_vocab(model) };
    let mut n_cur = batch.n_tokens;
    let mut n_decode = 0;
    let mut candidates: Vec<llama_token_data> = (0..n_vocab)
        .map(|token_id| llama_token_data {
            id: token_id,
            logit: 0.0,
            p: 0.0,
        })
        .collect();
    let eos = unsafe { llama_token_eos(model) };

    while n_cur <= N_LEN {
        // Sample the next token
        {
            let logits = unsafe { slice::from_raw_parts(llama_get_logits_ith(ctx, batch.n_tokens - 1), n_vocab as usize) };

            for (token_id, candidate) in candidates.iter_mut().enumerate() {
                candidate.id = token_id.try_into().unwrap();
                candidate.logit = logits[token_id];
                candidate.p = 0.0;
            }

            let mut candidates_p = llama_token_data_array {
                data: candidates.as_mut_ptr(),
                size: candidates.len(),
                sorted: false,
            };

            // sample the most likely token
            // Safety: Candidates outlives the call to this function.
            let new_token_id: llama_token = unsafe { llama_sample_token_greedy(ctx, &mut candidates_p) };

            tokens_list.push(new_token_id);

            // is it the end of the stream?
            if new_token_id == eos || n_cur == N_LEN {
                break;
            }

            print!("{}", token_to_piece(new_token_id, model));
            stdout().flush()?;

            // prepare the next batch (llama_batch_clear does this)
            batch.n_tokens = 0;

            // push this new token for next evaluation
            llama_batch_add(&mut batch, new_token_id, n_cur.try_into().unwrap(), true);

            n_decode += 1;
        }

        n_cur += 1;

        // evaluate the current batch wih the transformer model
        let ret = unsafe { llama_decode(ctx, batch) };
        if ret != 0 {
            return Err(format!("Failed to eval, return code {ret}").into());
        }
    }

    // Cleanup
    unsafe {
        llama_batch_free(batch);
        llama_free(ctx);
        llama_free_model(model);
        llama_backend_free();
    }

    // Check results (this should match the next part of the poem)
    assert_eq!(&tokens_list, &[
        1,
        1938,
        366,
        763,
        7933,
        29808,
        322,
        16366,
        29973,
        13,
        29902,
        437,
        451,
        763,
        963,
        29892,
        3685,
        29899,
        29902,
        29899,
        314,
        29889,
        13,
        29902,
        437,
        451,
        763,
        7933,
        29808,
        322,
        16366,
        29889,
        13,
    ]);

    Ok(())
}

/// Adapted from `llama.cpp/common/common.cpp`
fn token_to_piece(token: llama_token, model: *const llama_model) -> String {
    let mut buf = [0u8; 64];
    let n_tokens = unsafe { llama_token_to_piece(model, token, buf.as_mut_ptr() as *mut i8, buf.len().try_into().unwrap()) };
    buf[buf.len() - 1] = 0;
    if n_tokens < 0 {
        // should be unreachable
        panic!("Token `{token}` piece longer than max chars (64).");
    } else {
        return CStr::from_bytes_until_nul(&buf).unwrap().to_str().unwrap().into();
    }
}

/// Adapted from `llama.cpp/common/common.cpp`
fn llama_batch_add(batch: &mut llama_batch, token: llama_token, pos: usize, logits: bool) {
    assert!(batch.n_tokens <= BATCH_TOKENS as i32);

    let n_tokens:usize = batch.n_tokens.try_into().unwrap();

    unsafe {
        slice::from_raw_parts_mut(batch.token, BATCH_TOKENS)[n_tokens] = token;
        slice::from_raw_parts_mut(batch.pos, BATCH_TOKENS)[n_tokens] = pos.try_into().unwrap();
        slice::from_raw_parts_mut(batch.n_seq_id, BATCH_TOKENS)[n_tokens] = 1;
        let ids = slice::from_raw_parts_mut(batch.seq_id, BATCH_TOKENS)[n_tokens];
        slice::from_raw_parts_mut(ids, 1)[0] = 0;
        slice::from_raw_parts_mut(batch.logits, BATCH_TOKENS)[n_tokens] = logits as i8;
    };

    batch.n_tokens += 1;
}

/// Adapted from `llama.cpp/common/common.cpp`
fn tokenize(model: *const llama_model, text: &str, add_bos: bool) -> Vec<llama_token> {
    // upper limit for the number of tokens
    let mut n_tokens: i32 = (text.as_bytes().len() + if add_bos { 1 } else { 0 }).try_into().unwrap();
    let mut result = vec![0; n_tokens as usize];
    n_tokens = unsafe {
        llama_tokenize(
            model,
            text.as_bytes().as_ptr() as *const i8,
            text.len().try_into().unwrap(),
            result.as_mut_ptr(),
            result.len().try_into().unwrap(),
            add_bos,
            false,
        )
    };
    if n_tokens < 0 {
        result.resize((-n_tokens).try_into().unwrap(), 0);
        let check = unsafe {
            llama_tokenize(
                model,
                text.as_bytes().as_ptr() as *const i8,
                text.len().try_into().unwrap(),
                result.as_mut_ptr(),
                result.len().try_into().unwrap(),
                add_bos,
                false,
            )
        };
        assert_eq!(check, n_tokens)
    } else {
        result.resize(n_tokens.try_into().unwrap(), 0);
    }

    result
}
