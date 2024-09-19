import argparse

import ageml.datasets.synthetic_data as data_generator


def main():
    # Configure parser
    desc = "Generate synthetic data and save it to the data directory"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-o", "--output_dir", type=str, help="Output directory to save the synthetic data")

    args = parser.parse_args()
    # Generate synthetic data and save it to the data directory
    data_generator.generate_synthetic_data(save=True, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
