# 原始可编辑图的重建源

这些文件记录用户手工修改前的对象级重建，不是修改后PPT的生成脚本：

- `source.png`：原始图按相同比例缩小后的2100×880测量输入。
- `source_geometry.py`：原图的Matplotlib图形对象，读取仓库训练集第一条光谱。
- `author_manifest.py`：测量上述文字、形状、自由曲线和连线，写出唯一的对象清单。
- `manifest.json`：原始252个独立对象的坐标、样式与区域定位证据。
- `page_request.json`：原始自定义幻灯片尺寸与内容区域，已移除本机运行路径。

已安装NumPy、Matplotlib时可运行：

```powershell
python -X utf8 presentations/source/original_editable/author_manifest.py
```

这只更新本目录的 `manifest.json`。原PPT构建使用Codex的 `nature-image2ppt` 运行时，其构建器没有作为项目依赖复制入库；要在该运行时中重建，需遵循它的prepare、page build和渲染验证流程。通常直接编辑 [`../../LSTM网络结构图_可编辑版.pptx`](../../LSTM网络结构图_可编辑版.pptx)即可。
