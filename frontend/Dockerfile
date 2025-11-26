# Use an official Node.js runtime as a parent image
FROM node:20-alpine

# Set the working directory in the container
WORKDIR /app/frontend

# Copy package.json and package-lock.json to the working directory
COPY frontend/package*.json ./

# Install any needed packages specified in package.json
RUN npm install

# Copy the rest of the application's code into the container at /app
COPY frontend/. .

# Expose the port the app runs on
EXPOSE 3000

# Run the application
CMD ["npm", "start"]
