Run the following commands AFTER ACTIVATING THE PYTHON VIRTUAL ENVIRONMENT (in MacOS and LINUX):

pip install -r requirements.txt

Ensure Postgresql is installed and running.
Run the following command to create a database table:


psql -U postgres -d your_database -f database_setup.sql

Open retrieval.py and set your Database Credentials

DB_NAME = "your_db"
DB_USER = "your_user"
DB_PASSWORD = "your_password"
DB_HOST = "localhost"
DB_PORT = "5432"

Now run:

python retrieval.py

This will store the embeddings in Postgresql table using PGVector
Now start the server, using:

python app.py

Now the server is running at http://127.0.0.1:5000.
The endpoint to send questions will be http://127.0.0.1:5000/ask
To send questions to the server, use the following curl command:

curl -X POST "http://127.0.0.1:5000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the penalty for illegal dumping in Arcadia?"}'




Typical response time is 6-8 seconds.