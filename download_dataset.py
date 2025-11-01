import kagglehub

# Download latest version
path = kagglehub.dataset_download("karimabdulnabi/fruit-classification10-class")

print("Path to dataset files:", path)