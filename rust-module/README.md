# Wasm module for neutron plotter

## Components

### Autocomplete

The autocomplete system uses an extremely space-efficient LOUDS (Level-Order Unary Degree Sequence) Trie implementation.

As a LOUDS trie, it uses a bitvector to represent the parent-children hierarchy with as little memory as possible.

TODO: more explanation

### Plotter

A beam search A* search algorithm, which finds the approximated (not absolute) best route in a reasonable amount of time.

## Development

### Generating star data

Download systems_neutron from <https://spansh.co.uk/dumps>
and extract it into rust_module/systems_neutron.json

Also download "systems_1day and extract it as well.

Then run the following command:

```sh
cargo run --release
```

This will populate public/data/ with the generated star data in binary format.

## Usage

### Build with `wasm-pack`

```sh
wasm-pack build --target web
```

It will take a while on first run because it is compiling the rust standard library with WASM threading enabled.
