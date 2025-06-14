# Stage 1: Builder - Install dependencies
FROM python:3.10-slim-buster AS builder

WORKDIR /app

# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install dependencies using pip wheel for caching
RUN pip wheel --no-cache-dir --wheel-dir=/usr/src/app/wheels -r requirements.txt

# Stage 2: Runner - Create the final image
FROM python:3.10-slim-buster

WORKDIR /app

# Copy pre-built wheels from the builder stage
COPY --from=builder /usr/src/app/wheels /wheels

# Install packages from wheels
RUN pip install --no-cache-dir /wheels/* \
    && rm -rf /wheels # Clean up wheels after installation to reduce image size

# Copy the entire project
COPY . .

# Expose the Jupyter port (default 8888)
EXPOSE 8888

# Define an entrypoint script
ENTRYPOINT ["./start.sh"]

# Default command if no arguments are provided to the container
CMD ["jupyter"]