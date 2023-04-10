#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod test {
    use std::ffi::{CStr};
    use super::*;

    #[test]
    pub fn test_llama_print_system_info() {
        let system_info = unsafe {
            let raw_system_info = llama_print_system_info();
            CStr::from_ptr(raw_system_info)
        };
        println!("{}", system_info.to_string_lossy());
    }
}