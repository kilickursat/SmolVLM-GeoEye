#!/bin/bash
#
# SmolVLM-GeoEye Production Deployment Script
# ===========================================
#
# This script handles production deployment of SmolVLM-GeoEye
# including health checks, database migrations, and service orchestration.
#
# Usage: ./deploy-production.sh [environment]
#
# Author: SmolVLM-GeoEye Team
# Version: 3.1.0

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-production}
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"
BACKUP_DIR="./backups"
LOG_FILE="deployment_$(date +%Y%m%d_%H%M%S).log"

# Functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
    fi
    
    # Check environment file
    if [ ! -f "$ENV_FILE" ]; then
        error "Environment file $ENV_FILE not found"
    fi
    
    # Validate environment variables
    source "$ENV_FILE"
    
    if [ -z "${RUNPOD_API_KEY:-}" ]; then
        error "RUNPOD_API_KEY is not set"
    fi
    
    if [ -z "${RUNPOD_ENDPOINT_ID:-}" ]; then
        error "RUNPOD_ENDPOINT_ID is not set"
    fi
    
    if [ -z "${DB_PASSWORD:-}" ]; then
        error "DB_PASSWORD is not set"
    fi
    
    if [ -z "${SECRET_KEY:-}" ]; then
        warning "SECRET_KEY is not set - generating one"
        SECRET_KEY=$(openssl rand -hex 32)
        echo "SECRET_KEY=$SECRET_KEY" >> "$ENV_FILE"
    fi
    
    log "Prerequisites check completed"
}

# Create required directories
create_directories() {
    log "Creating required directories..."
    
    mkdir -p data logs backups nginx/ssl monitoring/prometheus monitoring/grafana/dashboards monitoring/grafana/datasources
    
    log "Directories created"
}

# Backup existing data
backup_data() {
    log "Backing up existing data..."
    
    BACKUP_NAME="backup_$(date +%Y%m%d_%H%M%S)"
    BACKUP_PATH="$BACKUP_DIR/$BACKUP_NAME"
    
    mkdir -p "$BACKUP_PATH"
    
    # Backup database if running
    if docker-compose ps db | grep -q "Up"; then
        log "Backing up database..."
        docker-compose exec -T db pg_dump -U postgres geotechnical | gzip > "$BACKUP_PATH/database.sql.gz"
    fi
    
    # Backup application data
    if [ -d "data" ] && [ "$(ls -A data)" ]; then
        log "Backing up application data..."
        tar -czf "$BACKUP_PATH/data.tar.gz" data/
    fi
    
    log "Backup completed: $BACKUP_PATH"
}

# Pull latest images
pull_images() {
    log "Pulling latest Docker images..."
    
    docker-compose pull
    
    log "Images pulled successfully"
}

# Build custom images
build_images() {
    log "Building custom Docker images..."
    
    docker-compose build --no-cache
    
    log "Images built successfully"
}

# Run database migrations
run_migrations() {
    log "Running database migrations..."
    
    # Start only the database service
    docker-compose up -d db
    
    # Wait for database to be ready
    log "Waiting for database to be ready..."
    sleep 10
    
    # Run migrations (if you have a migration script)
    # docker-compose run --rm app python manage.py migrate
    
    log "Migrations completed"
}

# Deploy services
deploy_services() {
    log "Deploying services..."
    
    # Stop existing services
    if docker-compose ps | grep -q "Up"; then
        log "Stopping existing services..."
        docker-compose down
    fi
    
    # Start all services
    log "Starting services..."
    docker-compose up -d
    
    # Wait for services to be ready
    log "Waiting for services to be ready..."
    sleep 30
    
    log "Services deployed"
}

# Health checks
run_health_checks() {
    log "Running health checks..."
    
    # Check main application
    if curl -f http://localhost:8501/health &> /dev/null; then
        log "✅ Main application is healthy"
    else
        error "Main application health check failed"
    fi
    
    # Check API
    if curl -f http://localhost:8000/health &> /dev/null; then
        log "✅ API service is healthy"
    else
        warning "API service health check failed"
    fi
    
    # Check database
    if docker-compose exec -T db pg_isready -U postgres &> /dev/null; then
        log "✅ Database is healthy"
    else
        error "Database health check failed"
    fi
    
    # Check Redis
    if docker-compose exec -T redis redis-cli ping | grep -q "PONG"; then
        log "✅ Redis is healthy"
    else
        warning "Redis health check failed"
    fi
    
    # Check RunPod connection
    if docker-compose exec -T app python -c "
from modules.config import get_config
from modules.smolvlm_client import EnhancedRunPodClient
config = get_config()
client = EnhancedRunPodClient(config)
health = client.health_check()
exit(0 if health['ready'] else 1)
" &> /dev/null; then
        log "✅ RunPod connection is healthy"
    else
        warning "RunPod connection check failed"
    fi
    
    log "Health checks completed"
}

# Configure monitoring
configure_monitoring() {
    log "Configuring monitoring..."
    
    # Create Prometheus configuration
    cat > monitoring/prometheus.yml <<EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'smolvlm-geoeye'
    static_configs:
      - targets: ['app:8501', 'api:8000']
EOF
    
    # Create Grafana datasource
    cat > monitoring/grafana/datasources/prometheus.yml <<EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF
    
    log "Monitoring configured"
}

# Configure nginx
configure_nginx() {
    log "Configuring Nginx..."
    
    # Create nginx configuration
    cat > nginx/nginx.conf <<'EOF'
events {
    worker_connections 1024;
}

http {
    upstream app {
        server app:8501;
    }
    
    upstream api {
        server api:8000;
    }
    
    server {
        listen 80;
        server_name _;
        
        location / {
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
        
        location /api {
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
EOF
    
    log "Nginx configured"
}

# Setup cron jobs
setup_cron_jobs() {
    log "Setting up cron jobs..."
    
    # Create cron job for backups
    cat > backup_cron.sh <<'EOF'
#!/bin/bash
cd /app
./deploy-production.sh backup_only
EOF
    
    chmod +x backup_cron.sh
    
    # Add to crontab (runs daily at 2 AM)
    # (crontab -l 2>/dev/null; echo "0 2 * * * /app/backup_cron.sh") | crontab -
    
    log "Cron jobs setup completed"
}

# Show deployment summary
show_summary() {
    log "Deployment Summary:"
    echo "===================="
    echo "Environment: $ENVIRONMENT"
    echo "Services deployed:"
    docker-compose ps
    echo ""
    echo "Access URLs:"
    echo "- Main Application: http://localhost:8501"
    echo "- API: http://localhost:8000/docs"
    echo "- Prometheus: http://localhost:9090"
    echo "- Grafana: http://localhost:3000 (admin/admin)"
    echo ""
    echo "Logs: docker-compose logs -f [service_name]"
    echo "===================="
}

# Backup only function
backup_only() {
    check_prerequisites
    backup_data
    exit 0
}

# Main deployment flow
main() {
    log "Starting SmolVLM-GeoEye deployment for $ENVIRONMENT environment"
    
    # Handle backup only mode
    if [ "${1:-}" = "backup_only" ]; then
        backup_only
    fi
    
    # Full deployment
    check_prerequisites
    create_directories
    backup_data
    configure_nginx
    configure_monitoring
    pull_images
    build_images
    run_migrations
    deploy_services
    run_health_checks
    setup_cron_jobs
    show_summary
    
    log "Deployment completed successfully! 🎉"
}

# Run main function
main "$@"
