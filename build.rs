use std::env;
use std::path::PathBuf;
use cmake::Config;

fn main() {
    let dst = Config::new("external/llama.cpp")
        .build_target("preinstall")
        .define("BUILD_SHARED_LIBS", "true")
        .define("LLAMA_ACCELERATE", "true")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .very_verbose(true)
        .build();

    println!("cargo:rustc-link-search=native={}/build", dst.display());
    println!("cargo:rustc-link-lib=dylib=llama");

    println!("cargo:rerun-if-changed=external/llama.cpp/llama.h");

    let bindings = bindgen::Builder::default()
        .header("external/llama.cpp/llama.h")
        .allowlist_function("llama_.*")
        .allowlist_type("llama_.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}