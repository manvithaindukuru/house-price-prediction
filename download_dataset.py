import kagglehub

path = kagglehub.dataset_download(
    "rishitaverma02/house-prices-advanced-regression-techniques"
)

print("Dataset Path:", path)