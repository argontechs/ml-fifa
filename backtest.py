"""Walk-forward backtest: tune on 2022-23, report on 2024-today. Writes data/backtest_report.json."""
from fifa import backtest_lib, data, evaluate

result = backtest_lib.run(report_path=data.DATA_DIR / "backtest_report.json")
print(f"\n=== TEST {result['test_span'][0]} .. {result['test_span'][1]} "
      f"(rho={result['rho']}, w_dc={result['w_dc']}) ===")
print("--- raw (uncalibrated) ---")
print(evaluate.format_card(result["raw_card"], gates=False))
print("--- calibrated (official) ---")
print(evaluate.format_card(result["test_card"]))
