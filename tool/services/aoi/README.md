# AOI 缺陷分类服务

- Tool：`pcba_defect_classification`
- 基础 URL：`http://127.0.0.1:8000`
- 健康检查：`GET /health`
- 分类接口：`POST /api/v1/classify`
- 详细接口：[docs/API.md](docs/API.md)

从项目根目录启动：

```powershell
conda run -n PCB_Agent python -m uvicorn tool.services.aoi.app.main:app --host 0.0.0.0 --port 8000
```
