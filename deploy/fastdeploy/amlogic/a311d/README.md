[English](README.md) | 简体中文

# PaddleDetection 检测模型在晶晨NPU上的部署方案部署方案—FastDeploy  

目前 FastDeploy 已经支持基于 Paddle Lite 部署 PP-YOLOE  量化模型到 A311D 上。

## 1. 说明  

晶晨A311D是一款先进的AI应用处理器。PaddleDetection支持通过FastDeploy在A311D上基于Paddle-Lite部署相关检测模型。**注意**：需要注意的是，芯原（verisilicon）作为 IP 设计厂商，本身并不提供实体SoC产品，而是授权其 IP 给芯片厂商，如：晶晨（Amlogic），瑞芯微（Rockchip）等。因此本文是适用于被芯原授权了 NPU IP 的芯片产品。只要芯片产品没有大副修改芯原的底层库，则该芯片就可以使用本文档作为 Paddle Lite 推理部署的参考和教程。在本文中，晶晨 SoC 中的 NPU 和 瑞芯微 SoC 中的 NPU 统称为芯原 NPU。目前支持如下芯片的部署：
- Amlogic A311D
- Amlogic C308X
- Amlogic S905D3

模型的量化和量化模型的下载请参考：[模型量化](../quantize/README.md)

## 2. 详细的部署示例

在 A311D 上只支持 C++ 的部署。

- [C++部署](cpp)