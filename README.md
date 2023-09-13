# llama-cpp-sys

This is a binding of [llama.cpp](https://github.com/ggerganov/llama.cpp) for rust.

For a higher-level API, see https://github.com/shadowmint/llama-cpp-rs

## Build

    cargo build

## Run example

Put your models in the `models` folder; the test expects a file in the path:

    models/model.gguf

Then run:

    cargo test --release --test "test_generate" -- --nocapture

You should see output like:

```
running 1 test
llama_model_load: loading model from 'models/13B/model.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 5120
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 40
llama_model_load: n_layer = 40
llama_model_load: n_rot   = 128
llama_model_load: f16     = 2
llama_model_load: n_ff    = 13824
llama_model_load: n_parts = 2
llama_model_load: type    = 2
llama_model_load: ggml map size = 7759.83 MB
llama_model_load: ggml ctx size = 101.25 KB
llama_model_load: mem required  = 9807.93 MB (+ 1608.00 MB per state)
llama_model_load: loading tensors from 'models/13B/model.bin'
llama_model_load: model size =  7759.39 MB / num tensors = 363
llama_init_from_file: kv self size  =  400.00 MB
loaded model
prompt length: 20
model context length: 512
prompt:
 hello, my name is dave and I live in Oz. Where do I live?

generating...

! I don't know, but it feels really good!!
I am an Aussie, not from the bush, although you might think so after hearing me talk, but I am a proud Aussie none-the-less!!
When my family and I decided to immigrate to Australia, we all knew that it was going to be hard at times; however, if you are willing to put in the effort then you can make it anywhere, even in this great land of Oz.
We left our old home behind, selling everything that wouldn't fit into a suitcase and moving as far

test main ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 14.10s
```
