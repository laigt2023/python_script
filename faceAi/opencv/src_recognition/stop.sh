#!/bin/bash

# 定义要搜索的jar包名称
jar_name="ai-server.py"

# 根据jar包名称搜索java进程
pid=$(ps -ef | grep "$jar_name" | grep -v grep | awk '{print $2}')

if [ -n $pid ]; then
  # 杀掉指定名称的java进程
  kill -9 $pid
  echo "已成功杀掉进程 $pid"
else
  echo "未找到与 $jar_name 相关的进程"
fi
