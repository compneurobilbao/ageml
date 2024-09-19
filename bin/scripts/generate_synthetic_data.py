import ageml.datasets.synthetic_data as data_generator

if __name__ == "__main__":
    # Generate synthetic data and save it to the data directory
    data_generator.generate_synthetic_data(save=True)
