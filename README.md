# [ZeroGrads](https://mfischer-ucl.github.io/zerograds/) unofficial implementation in Rust with Candle

This repository is a project designed to verify the performance of the ZeroGrads algorithm implemented in Rust using
the `candle` library. ZeroGrads utilizes a surrogate function-based gradient descent method, effective particularly for
optimizing complex functions where gradient information is not directly accessible.

## Project Objective

This project evaluates the computational speed achievable using the ZeroGrads method on the `candle` library, utilizing
a simple 1000-dimensional quadratic function referenced from the publicly available ZeroGrads Colab notebook. If
opportunities arise, we will add other complex examples as well.

## How to Use

### Prerequisites

Ensure that you have the Rust development environment set up before running this program. Rust can be easily installed
via [rustup](https://rustup.rs/).

### Execution

To run the program, use the following command. It is recommended to execute it in release mode to generate an optimized
binary and enhance execution speed.

```bash
cargo run --release
```

## Contributions

This project is open-source, and contributions such as improvement suggestions or additional features via pull requests
are always welcome. Please use GitHub issues for specific bug reports or feature requests.

## License

This project is published under the [MIT License](LICENSE). For details, refer to the license file.
