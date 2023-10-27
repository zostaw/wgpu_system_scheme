.PHONY: all

all: install build run

install:
	cargo install -f wasm-bindgen-cli

build: 
	RUSTFLAGS=--cfg=web_sys_unstable_apis cargo build --target wasm32-unknown-unknown
	wasm-bindgen --out-dir generated --web target/wasm32-unknown-unknown/debug/system_scheme.wasm

run: 
	python -m http.server -d generated &
