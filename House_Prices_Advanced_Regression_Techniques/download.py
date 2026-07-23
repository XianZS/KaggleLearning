# The following code will only execute
# successfully when compression is complete
import os
import kagglehub

# Download latest version
os.environ["kagglehub_cache"] = "./data"
path = kagglehub.competition_download("house-prices-advanced-regression-techniques")

print("Path to competition files:", path)
