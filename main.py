"""Точка входа приложения."""
from ui.theme import setup_appearance
from ui.app import LogisticsApp


def main():
    setup_appearance()
    app = LogisticsApp()
    app.mainloop()


if __name__ == "__main__":
    main()