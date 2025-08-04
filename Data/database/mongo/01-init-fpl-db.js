// MongoDB initialization script for FPL Team Picker
// This script creates the database and sets up initial indexes

print('🚀 Initializing FPL Team Picker MongoDB database...');

db = db.getSiblingDB('fpl_team_picker');

// Create collections and indexes
print('📊 Creating players collection and indexes...');
db.createCollection('players');
db.players.createIndex({ 'id': 1 }, { unique: true });
db.players.createIndex({ 'element_type': 1 });
db.players.createIndex({ 'team': 1 });
db.players.createIndex({ 'web_name': 1 });

print('⚽ Creating teams collection and indexes...');
db.createCollection('teams');
db.teams.createIndex({ 'id': 1 }, { unique: true });
db.teams.createIndex({ 'short_name': 1 });

print('🏟️ Creating fixtures collection and indexes...');
db.createCollection('fixtures');
db.fixtures.createIndex({ 'id': 1 }, { unique: true });
db.fixtures.createIndex({ 'gameweek': 1, 'season': 1 });
db.fixtures.createIndex({ 'team_h': 1 });
db.fixtures.createIndex({ 'team_a': 1 });
db.fixtures.createIndex({ 'season': 1 });

print('✅ FPL Team Picker database initialized successfully!');
print('📈 Database collections and indexes created');
print('🔗 Connect using: mongodb://admin:fplteampicker2025@localhost:27017/fpl_team_picker');
