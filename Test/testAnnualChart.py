import win32com.client
from datetime import date


def getAnnualChart(code):
    instStockChart = win32com.client.Dispatch("CpSysDib.StockChart")

    instStockChart.SetInputValue(0, code)
    instStockChart.SetInputValue(1, ord('2'))
    instStockChart.SetInputValue(4, 365)
    instStockChart.SetInputValue(5, (0, 5))
    instStockChart.SetInputValue(6, ord('D'))
    instStockChart.SetInputValue(9, ord('1'))

    instStockChart.BlockRequest()

    numData = instStockChart.GetHeaderValue(3)
    print(numData)

    date_price_table = {
        'date': [date(
            instStockChart.GetDataValue(0, i) // 10000, 
            instStockChart.GetDataValue(0, i) // 100 % 100, 
            instStockChart.GetDataValue(0, i) % 100)
            for i in range(numData)], 
        'price': [instStockChart.GetDataValue(1, i) for i in range(numData)]}

    return date_price_table

#print(getAnnualChart("A001340"))
