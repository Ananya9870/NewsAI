# Use the official Python base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Create cache directories with proper permissions
RUN mkdir -p /app/cache/huggingface /app/cache/transformers /app/cache/sentence_transformers
RUN chmod -R 777 /app/cache

# Set environment variables for cache locations
ENV HF_HOME=/app/cache/huggingface
ENV TRANSFORMERS_CACHE=/app/cache/transformers
ENV SENTENCE_TRANSFORMERS_HOME=/app/cache/sentence_transformers


# Copy the model and code files
COPY . .


# Create directory for ChromaDB
RUN mkdir -p ./chroma_db
RUN chmod -R 777 ./chroma_db

# Expose the port FastAPI will run on
EXPOSE 7860

# Command to run the FastAPI app using Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]