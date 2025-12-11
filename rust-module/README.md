# Wasm module for neutron plotter

## Generating star data

Download systems_neutron from https://spansh.co.uk/dumps
and extract it into rust_module/systems_neutron.json

Then run the following command:

```
cargo run --release
```

This will populate public/data/ with the generated star data in binary format.

## 🚴 Usage

### 🛠️ Build with `wasm-pack build`

```
wasm-pack build
```
