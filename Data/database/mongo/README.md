# MongoDB Setup Guide for FPL Team Picker

This guide will help you migrate from JSON files to MongoDB for better data management and performance.

## 📋 Prerequisites

- Python 3.8+
- Docker and Docker Compose (recommended)
- OR MongoDB installed locally

## 🐳 Option 1: Docker Setup (Recommended)

### 1. Start MongoDB with Docker Compose

```bash
# Start MongoDB and Mongo Express (web UI)
docker-compose -f docker-compose.mongodb.yml up -d

# Verify containers are running
docker ps
```

This will start:
- **MongoDB** on `localhost:27017`
- **Mongo Express** (web UI) on `localhost:8081`

### 2. Web UI Access

- **Mongo Express URL**: http://localhost:8081
- **Username**: `fpl`
- **Password**: `teampicker`

## 🔧 Option 2: Local MongoDB Installation

### macOS (using Homebrew)
```bash
# Install MongoDB
brew tap mongodb/brew
brew install mongodb-community

# Start MongoDB
brew services start mongodb-community

# Verify MongoDB is running
brew services list | grep mongodb
```

### Ubuntu/Debian
```bash
# Import the public key
wget -qO - https://www.mongodb.org/static/pgp/server-7.0.asc | sudo apt-key add -

# Add MongoDB repository
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list

# Install MongoDB
sudo apt-get update
sudo apt-get install -y mongodb-org

# Start MongoDB
sudo systemctl start mongod
sudo systemctl enable mongod
```

## 📦 Python Package Installation

```bash
# Install MongoDB Python packages
pip install -r requirements-mongo.txt

# Or install individually
pip install pymongo motor python-dotenv
```

## ⚙️ Environment Configuration

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Edit `.env` file for your setup:

### For Docker setup:
```bash
MONGODB_URI=mongodb://admin:fplteampicker2025@localhost:27017/fpl_team_picker
MONGODB_DATABASE=fpl_team_picker
MONGODB_PLAYERS_COLLECTION=players
MONGODB_TEAMS_COLLECTION=teams
MONGODB_FIXTURES_COLLECTION=fixtures
```

### For local MongoDB (no authentication):
```bash
MONGODB_URI=mongodb://localhost:27017/fpl_team_picker
MONGODB_DATABASE=fpl_team_picker
MONGODB_PLAYERS_COLLECTION=players
MONGODB_TEAMS_COLLECTION=teams
MONGODB_FIXTURES_COLLECTION=fixtures
```

## 📊 Data Migration

### 1. Test MongoDB Connection

```bash
cd Data/database/mongo
python fpl_mongo_client.py
```

You should see:
```
✅ Connected to MongoDB: fpl_team_picker
📊 Database Statistics:
   players_count: 0
   teams_count: 0
   fixtures_count: 0
   database_size_mb: 0.0
   index_size_mb: 0.0
🔌 MongoDB connection closed
```

### 2. Migrate JSON Data to MongoDB

```bash
cd Data/database/mongo

# Run migration (keeps existing data)
python migrate_json_to_mongo.py

# Or reset and migrate (WARNING: deletes existing data)
python migrate_json_to_mongo.py --reset
```

Expected output:
```
🚀 Starting complete migration...
🔍 Validating JSON files...
   ✅ players.json found (2.84 MB)
   ✅ teams.json found (0.01 MB)
   ✅ fixtures.json found (0.12 MB)
📈 Creating database indexes...
✅ Database indexes created
⚽ Migrating teams...
📊 Loaded 20 records from teams.json
✅ Bulk upserted 20 teams
✅ Migrated 20 teams
👥 Migrating players...
📊 Loaded 639 records from players.json
✅ Bulk upserted 639 players
✅ Migrated 639 players
🏟️ Migrating fixtures...
📊 Loaded 380 records from fixtures.json
✅ Bulk upserted 380 fixtures
✅ Migrated 380 fixtures
📊 Migration Complete! Database Statistics:
   players_count: 639
   teams_count: 20
   fixtures_count: 380
   database_size_mb: 0.38
   index_size_mb: 0.02
🎉 All operations completed successfully!
```

### 3. Verify Migration

```bash
python migrate_json_to_mongo.py --verify-only
```

## 🔄 Update Python Scripts

### Automatic Update (Recommended)

```bash
cd Data/database/mongo

# Dry run to see what would change
python update_scripts_for_mongo.py --dry-run

# Apply changes (creates backups)
python update_scripts_for_mongo.py
```

### Manual Update

For scripts not caught by the automatic updater, replace:

```python
# OLD: JSON file loading
with open('Data/database/players.json', 'r') as f:
    players_data = json.load(f)

with open('Data/database/teams.json', 'r') as f:
    teams_data = json.load(f)
```

With:

```python
# NEW: MongoDB loading with JSON fallback
from Data.database.mongo.mongo_data_loader import load_players_data, load_teams_data

players_data = load_players_data()
teams_data = load_teams_data()
```

## 🧪 Testing the Setup

### 1. Test Data Loading

```bash
cd Data/database/mongo
python mongo_data_loader.py
```

### 2. Test with Existing Scripts

```bash
# Test prediction generation
cd Data/predictions_2025_26
python generate_predictions.py 1

# Test model training (should now use MongoDB)
cd Data/models/expected_goals
python train_expected_goals_model.py
```

## 🔍 MongoDB Operations

### Basic Queries

```python
from Data.database.mongo.fpl_mongo_client import FPLMongoClient

client = FPLMongoClient()

# Get all players
players = client.get_all_players()

# Get players by team
arsenal_players = client.get_players_by_team(1)  # Arsenal

# Get players by position
goalkeepers = client.get_players_by_position(1)  # GK

# Get fixtures by gameweek
gw1_fixtures = client.get_fixtures_by_gameweek(1)

client.disconnect()
```

### Using Mongo Express (Web UI)

1. Open http://localhost:8081
2. Login with `fpl` / `teampicker`
3. Navigate to `fpl_team_picker` database
4. Explore collections: `players`, `teams`, `fixtures`

## 📈 Performance Benefits

- **Faster queries**: Indexed searches vs. loading entire JSON files
- **Memory efficient**: Load only needed data
- **Concurrent access**: Multiple scripts can access data simultaneously
- **Data integrity**: Built-in validation and constraints
- **Scalability**: Easy to add new data sources and collections

## 🛠️ Podman on macOS: Restart & Start Containers

If you are using Podman on macOS and encounter connection or startup issues, follow these steps to safely restart your MongoDB containers **without losing data**:

### 1. Restart the Podman machine
```bash
podman machine stop
podman machine start
```

### 2. Start your MongoDB containers (no data loss)
```bash
podman start fpl-mongodb fpl-mongo-express
```

### 3. Check container status
```bash
podman ps -a
```

If you still have issues, a full system reboot will clear any stuck Podman/VM processes and allow you to start cleanly.

---
## 🔧 Troubleshooting

### Connection Issues

```bash
# Check if MongoDB is running
docker ps  # For Docker setup
brew services list | grep mongodb  # For macOS local install

# Test connection manually
python -c "from pymongo import MongoClient; print(MongoClient('mongodb://localhost:27017').admin.command('ismaster'))"
```

### Import Errors

```bash
# Ensure packages are installed
pip install pymongo python-dotenv

# Check Python path
python -c "import sys; print(sys.path)"
```

### Data Issues

```bash
# Re-run migration with reset
cd Data/database/mongo
python migrate_json_to_mongo.py --reset

# Check JSON files exist
ls -la ../
```

## 🚀 Next Steps

1. **Update TODO.md**: Mark "Change the JSON files to a proper data store" as complete
2. **API Integration**: Update the .NET API to connect to MongoDB
3. **Real-time Updates**: Set up data refresh workflows
4. **Monitoring**: Add logging and monitoring for database operations

## 🔄 Rollback Plan

If you need to rollback to JSON files:

1. **Restore backups**: Use the `.backup` files created during script updates
2. **Revert imports**: Change MongoDB imports back to JSON file loading
3. **Stop MongoDB**: `docker-compose -f docker-compose.mongodb.yml down`

The JSON files remain unchanged, so rollback is always possible.
