FROM python:3.11-slim

#working directory in the container
WORKDIR /app

#dependencies file to the working directory
COPY requirements.txt .

# Install needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define the command to run your app
CMD ["streamlit", "run", "app.py"]