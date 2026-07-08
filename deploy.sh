#!/usr/bin/env bash
# ============================================================
# OpenINTJ 一键部署脚本
# 用法：
#   chmod +x deploy.sh
#   ./deploy.sh                          # 使用已有 .env 文件
#   ./deploy.sh --api-key YOUR_KEY       # 通过参数传入 API Key
# ============================================================

set -euo pipefail

# ---------- 颜色输出 ----------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ---------- 常量 ----------
PROJECT_NAME="openintj"
IMAGE_NAME="openintj:latest"
CONTAINER_NAME="openintj"
APP_PORT=8000
HOST_PORT=8000
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"

# ---------- 参数解析 ----------
API_KEY=""
USE_COMPOSE=false
SKIP_BUILD=false
SETUP_NGINX=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --compose)
            USE_COMPOSE=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --nginx)
            SETUP_NGINX=true
            shift
            ;;
        --help|-h)
            echo "用法: ./deploy.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --api-key KEY    指定混元 API Key（自动写入 .env 文件）"
            echo "  --compose        使用 docker-compose 部署（默认使用 docker run）"
            echo "  --skip-build     跳过镜像构建步骤"
            echo "  --nginx          安装并配置 Nginx 反向代理（80→8000）"
            echo "  --help, -h       显示帮助信息"
            exit 0
            ;;
        *)
            warn "未知参数: $1"
            shift
            ;;
    esac
done

# ============================================================
# 步骤 1：检查 Docker 环境
# ============================================================
info "步骤 1/6：检查 Docker 环境..."

if ! command -v docker &> /dev/null; then
    error "Docker 未安装！请先安装 Docker：\n  curl -fsSL https://get.docker.com | bash"
fi

if ! docker info &> /dev/null; then
    error "Docker 守护进程未运行！请启动 Docker：\n  sudo systemctl start docker"
fi

DOCKER_VERSION=$(docker --version | awk '{print $3}' | tr -d ',')
success "Docker 已就绪 (版本: ${DOCKER_VERSION})"

# 检查 docker-compose（如果使用 compose 模式）
if [ "$USE_COMPOSE" = true ]; then
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    elif docker compose version &> /dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    else
        error "docker-compose 未安装！请安装后重试，或不使用 --compose 参数"
    fi
    success "Docker Compose 已就绪 (${COMPOSE_CMD})"
fi

# ============================================================
# 步骤 2：检查并配置 .env 文件
# ============================================================
info "步骤 2/6：检查环境变量配置..."

# 如果通过参数传入了 API Key，写入 .env 文件
if [ -n "$API_KEY" ]; then
    if [ -f "$ENV_FILE" ]; then
        # 更新已有 .env 文件中的 API Key
        if grep -q "^HUNYUAN_API_KEY=" "$ENV_FILE"; then
            sed -i "s|^HUNYUAN_API_KEY=.*|HUNYUAN_API_KEY=${API_KEY}|" "$ENV_FILE"
        else
            echo "HUNYUAN_API_KEY=${API_KEY}" >> "$ENV_FILE"
        fi
    else
        # 从 .env.example 复制并设置 API Key
        if [ -f "${SCRIPT_DIR}/.env.example" ]; then
            cp "${SCRIPT_DIR}/.env.example" "$ENV_FILE"
            sed -i "s|^HUNYUAN_API_KEY=.*|HUNYUAN_API_KEY=${API_KEY}|" "$ENV_FILE"
        else
            # 直接创建 .env 文件
            cat > "$ENV_FILE" <<EOF
HUNYUAN_API_KEY=${API_KEY}
HUNYUAN_BASE_URL=https://api.hunyuan.cloud.tencent.com/v1
HUNYUAN_MODEL=hunyuan-turbos-latest
HUNYUAN_VISION_MODEL=hunyuan-vision
EOF
        fi
    fi
    success "API Key 已写入 .env 文件"
fi

# 检查 .env 文件是否存在
if [ ! -f "$ENV_FILE" ]; then
    warn ".env 文件不存在，将从 .env.example 创建..."
    if [ -f "${SCRIPT_DIR}/.env.example" ]; then
        cp "${SCRIPT_DIR}/.env.example" "$ENV_FILE"
        warn "请编辑 .env 文件填入你的 HUNYUAN_API_KEY，然后重新运行此脚本"
        warn "  vim ${ENV_FILE}"
        exit 1
    else
        error ".env.example 文件也不存在，请先创建 .env 文件"
    fi
fi

# 检查 API Key 是否已配置
if grep -q "^HUNYUAN_API_KEY=$" "$ENV_FILE" || grep -q "^HUNYUAN_API_KEY=your_hunyuan_api_key_here$" "$ENV_FILE"; then
    warn "HUNYUAN_API_KEY 未配置，系统将以 Mock 模式运行"
    warn "如需使用真实 LLM，请设置 API Key：./deploy.sh --api-key YOUR_KEY"
else
    success "环境变量配置已就绪"
fi

# ============================================================
# 步骤 3：检查端口占用
# ============================================================
info "步骤 3/6：检查端口占用..."

if ss -tlnp 2>/dev/null | grep -q ":${HOST_PORT} " || \
   netstat -tlnp 2>/dev/null | grep -q ":${HOST_PORT} "; then
    # 检查是否是我们自己的容器占用
    EXISTING=$(docker ps --filter "name=${CONTAINER_NAME}" --format "{{.ID}}" 2>/dev/null || true)
    if [ -n "$EXISTING" ]; then
        info "端口 ${HOST_PORT} 被现有 ${CONTAINER_NAME} 容器占用，将在后续步骤中替换"
    else
        error "端口 ${HOST_PORT} 已被其他进程占用！请释放端口后重试：\n  sudo lsof -i :${HOST_PORT}"
    fi
else
    success "端口 ${HOST_PORT} 可用"
fi

# ============================================================
# 步骤 4：构建 Docker 镜像
# ============================================================
if [ "$SKIP_BUILD" = true ]; then
    info "步骤 4/6：跳过镜像构建（--skip-build）"
else
    info "步骤 4/6：构建 Docker 镜像..."
    cd "$SCRIPT_DIR"

    if docker build -t "$IMAGE_NAME" . ; then
        success "镜像构建成功: ${IMAGE_NAME}"
    else
        error "镜像构建失败！请检查 Dockerfile 和依赖配置"
    fi
fi

# ============================================================
# 步骤 5：停止并清理旧容器
# ============================================================
info "步骤 5/6：清理旧容器..."

# 停止旧容器
OLD_CONTAINER=$(docker ps -aq --filter "name=${CONTAINER_NAME}" 2>/dev/null || true)
if [ -n "$OLD_CONTAINER" ]; then
    info "停止旧容器: ${OLD_CONTAINER}"
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
    success "旧容器已清理"
else
    info "无旧容器需要清理"
fi

# 清理悬空镜像（释放磁盘空间）
DANGLING=$(docker images -f "dangling=true" -q 2>/dev/null || true)
if [ -n "$DANGLING" ]; then
    docker rmi $DANGLING 2>/dev/null || true
    info "已清理悬空镜像"
fi

# ============================================================
# 步骤 6：启动新容器
# ============================================================
info "步骤 6/6：启动新容器..."

if [ "$USE_COMPOSE" = true ]; then
    # Docker Compose 模式
    cd "$SCRIPT_DIR"
    $COMPOSE_CMD up -d
    success "容器已通过 Docker Compose 启动"
else
    # Docker Run 模式
    docker run -d \
        --name "$CONTAINER_NAME" \
        --restart unless-stopped \
        --env-file "$ENV_FILE" \
        -p "${HOST_PORT}:${APP_PORT}" \
        --health-cmd "curl -f http://localhost:${APP_PORT}/api/health || exit 1" \
        --health-interval 30s \
        --health-timeout 10s \
        --health-retries 3 \
        --health-start-period 15s \
        --log-opt max-size=50m \
        --log-opt max-file=3 \
        "$IMAGE_NAME"

    success "容器已启动: ${CONTAINER_NAME}"
fi

# ============================================================
# 步骤 7（可选）：安装并配置 Nginx 反向代理
# ============================================================
if [ "$SETUP_NGINX" = true ]; then
    info "步骤 7/7：配置 Nginx 反向代理..."

    NGINX_CONF_SRC="${SCRIPT_DIR}/nginx.conf"
    NGINX_CONF_DST="/etc/nginx/sites-available/openintj"
    NGINX_CONF_LINK="/etc/nginx/sites-enabled/openintj"

    # 检查 nginx.conf 源文件是否存在
    if [ ! -f "$NGINX_CONF_SRC" ]; then
        error "nginx.conf 文件不存在！请确保项目根目录下有 nginx.conf 文件"
    fi

    # 检查 Nginx 是否已安装
    if ! command -v nginx &> /dev/null; then
        info "Nginx 未安装，正在安装..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update -qq
            sudo apt-get install -y -qq nginx
        elif command -v yum &> /dev/null; then
            sudo yum install -y -q nginx
        else
            error "无法自动安装 Nginx，请手动安装后重试"
        fi
        success "Nginx 安装完成"
    else
        success "Nginx 已安装 ($(nginx -v 2>&1 | awk -F/ '{print $2}'))"
    fi

    # 备份已有配置（如果存在）
    if [ -f "$NGINX_CONF_DST" ]; then
        BACKUP="${NGINX_CONF_DST}.bak.$(date +%Y%m%d%H%M%S)"
        sudo cp "$NGINX_CONF_DST" "$BACKUP"
        info "已备份旧配置: ${BACKUP}"
    fi

    # 复制配置文件
    sudo cp "$NGINX_CONF_SRC" "$NGINX_CONF_DST"

    # 创建软链接到 sites-enabled（如果目录存在）
    if [ -d "/etc/nginx/sites-enabled" ]; then
        sudo rm -f "$NGINX_CONF_LINK"
        sudo ln -s "$NGINX_CONF_DST" "$NGINX_CONF_LINK"

        # 移除默认配置（避免冲突）
        if [ -f "/etc/nginx/sites-enabled/default" ]; then
            sudo rm -f /etc/nginx/sites-enabled/default
            info "已移除 Nginx 默认配置"
        fi
    elif [ -d "/etc/nginx/conf.d" ]; then
        # 部分系统使用 conf.d 目录
        sudo cp "$NGINX_CONF_SRC" "/etc/nginx/conf.d/openintj.conf"
    fi

    # 测试 Nginx 配置
    if sudo nginx -t 2>/dev/null; then
        success "Nginx 配置语法检查通过"
    else
        error "Nginx 配置语法错误！请检查 nginx.conf 文件"
    fi

    # 重载 Nginx
    if systemctl is-active --quiet nginx 2>/dev/null; then
        sudo systemctl reload nginx
        success "Nginx 已重载"
    else
        sudo systemctl start nginx
        sudo systemctl enable nginx 2>/dev/null || true
        success "Nginx 已启动并设为开机自启"
    fi

    info "Nginx 反向代理已配置：80 → ${HOST_PORT}"
else
    info "跳过 Nginx 配置（如需配置，请添加 --nginx 参数）"
fi

# ============================================================
# 部署完成 —— 健康检查
# ============================================================
echo ""
info "等待服务启动..."
sleep 3

# 健康检查（最多等待 30 秒）
MAX_RETRIES=10
RETRY_INTERVAL=3
for i in $(seq 1 $MAX_RETRIES); do
    if curl -sf "http://localhost:${HOST_PORT}/api/health" > /dev/null 2>&1; then
        echo ""
        echo -e "${GREEN}============================================================${NC}"
        echo -e "${GREEN}  ✅ OpenINTJ 部署成功！${NC}"
        echo -e "${GREEN}============================================================${NC}"
        echo ""
        echo -e "  🌐 访问地址:  ${CYAN}http://localhost:${HOST_PORT}${NC}"
        echo -e "  🔍 健康检查:  ${CYAN}http://localhost:${HOST_PORT}/api/health${NC}"
        echo -e "  🤖 LLM 状态:  ${CYAN}http://localhost:${HOST_PORT}/api/llm/status${NC}"
        echo ""
        echo -e "  📋 常用命令:"
        echo -e "     查看日志:  docker logs -f ${CONTAINER_NAME}"
        echo -e "     重启服务:  docker restart ${CONTAINER_NAME}"
        echo -e "     停止服务:  docker stop ${CONTAINER_NAME}"
        echo ""

        # 显示 LLM 状态
        LLM_STATUS=$(curl -sf "http://localhost:${HOST_PORT}/api/llm/status" 2>/dev/null || echo "{}")
        if echo "$LLM_STATUS" | grep -q '"mode":"live"'; then
            echo -e "  🟢 LLM 模式: ${GREEN}真实模式 (Live)${NC} — 已连接腾讯混元大模型"
        else
            echo -e "  🟡 LLM 模式: ${YELLOW}模拟模式 (Mock)${NC} — 请配置 HUNYUAN_API_KEY 启用真实 LLM"
        fi
        echo ""
        exit 0
    fi
    info "等待服务就绪... (${i}/${MAX_RETRIES})"
    sleep $RETRY_INTERVAL
done

# 健康检查失败
warn "服务未在预期时间内就绪，请手动检查："
echo "  docker logs ${CONTAINER_NAME}"
echo "  docker ps -a --filter name=${CONTAINER_NAME}"
exit 1
