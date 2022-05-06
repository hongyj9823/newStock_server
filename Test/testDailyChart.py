import win32com.client
from datetime import time

instStockChart = win32com.client.Dispatch("CpSysDib.StockChart")
instCodeMgr = win32com.client.Dispatch("CpUtil.CpCodeMgr")

instStockChart.SetInputValue(0, "A003540")
instStockChart.SetInputValue(1, ord('1'))
instStockChart.SetInputValue(3, 20220328)
instStockChart.SetInputValue(5, (0, 1, 5))
instStockChart.SetInputValue(6, ord('m'))
instStockChart.SetInputValue(9, ord('1'))

while True:
    instStockChart.BlockRequest()

    numData = instStockChart.GetHeaderValue(3)

    times = [time(instStockChart.GetDataValue(1, i) // 100,
        instStockChart.GetDataValue(1, i) % 100) for i in range(numData)]

    prices = [instStockChart.GetDataValue(2, i) for i in range(numData)]

    print(numData)
    print(times[-1])
    print(prices[-1])
    #chart = [[instStockChart.GetDataValue(0, i), instStockChart.GetDataValue(1, i)] for i in range(numData)]

    name = instCodeMgr.CodeToName("A003540")

    for timestamp, price in zip(times, prices):
        print(f"time:{timestamp}, price:{price}")
