# 端到端测试

先分别启动 AOI、回流焊和 SPI 服务，再从项目根目录执行：

```powershell
conda run -n PCB_Agent python -m tool.tests.e2e.run_e2e `
  --image "<测试图片路径>" `
  --report "tool/tests/e2e/artifacts/latest_report.json"
```

测试内容包括三项健康检查、5 个 Tool 成功调用、5 个 Tool 参数校验错误 Schema，以及两个优化 Tool 的再次预测验证。

如果正式端口临时被其他程序占用，可用 `start_service.ps1 -Port <端口>` 启动，并向测试命令传入 `--aoi-url`、`--reflow-url` 或 `--spi-url`；正式地址约定不因此改变。

也可以让测试脚本临时启动并在结束后停止三个服务：

```powershell
.\tool\tests\e2e\run_with_services.ps1
```

当前机器的 `8000` 被 NeatReader 占用，因此该脚本默认仅在测试期间把 AOI 放到 `18000`；正式约定仍为 `8000`。
