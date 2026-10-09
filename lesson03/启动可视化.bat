@echo off
chcp 65001 >nul
title 订单簿可视化 - 601688 华泰证券
cd /d "%~dp0"
echo ============================================
echo   订单簿交互式可视化 (真实数据)
echo   启动本地服务器: http://127.0.0.1:8765
echo   关闭本窗口即停止服务
echo ============================================
start "" http://127.0.0.1:8765/orderbook_vis.html
py -m http.server 8765 --bind 127.0.0.1
