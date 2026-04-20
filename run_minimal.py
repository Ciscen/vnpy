"""
VeighNa macOS 启动脚本

macOS 上 CTP/IB 等网关有平台兼容性限制，本脚本仅加载应用插件，
展示完整的 GUI 框架（策略引擎、回测器、数据管理器）。

如需连接实盘交易接口，请在 Linux/Windows 上运行 examples/veighna_trader/run.py。
"""
from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine
from vnpy.trader.ui import MainWindow, create_qapp

from vnpy_ctastrategy import CtaStrategyApp
from vnpy_ctabacktester import CtaBacktesterApp
from vnpy_datamanager import DataManagerApp


def main() -> None:
    qapp = create_qapp()

    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)

    main_engine.add_app(CtaStrategyApp)
    main_engine.add_app(CtaBacktesterApp)
    main_engine.add_app(DataManagerApp)

    main_window = MainWindow(main_engine, event_engine)
    main_window.showMaximized()

    qapp.exec()


if __name__ == "__main__":
    main()
