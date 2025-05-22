import pandas as pd
import numpy as np

def generate_student_data(num_samples=500, random_state=42):
    np.random.seed(random_state)
    data = pd.DataFrame({
        'hours_studied': np.random.normal(5, 2, num_samples),
        'attendance': np.random.randint(60, 100, num_samples),
        'participation': np.random.randint(1, 6, num_samples),
        'final_grade': np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6])
    })

    data = data[data['hours_studied'] >= 0]  # Remove negative values
    return data

if __name__ == "__main__":
    df = generate_student_data()
    df.to_csv('data/simulated_student_data.csv', index=False)
    print("Data saved to data/simulated_student_data.csv")
