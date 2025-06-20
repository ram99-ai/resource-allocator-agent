# Db connection credentials
hostname= "localhost"
user="root"
port=5432
pwd="RM@123"
owner="postgres"

DATABASE_CONFIG = {
    "employees": {"dbname": "employees", "user": user, "password": pwd, "host": hostname, "port": port, "owner":owner}
}

def discover_databases():
    return DATABASE_CONFIG
