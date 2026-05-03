FROM python:3.11.12-slim

WORKDIR /app

# Copy dependency list and install packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Run the training and prediction script
CMD ["python", "源码/train_and_predict.py"]
