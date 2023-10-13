import sys
from flask import Flask, render_template
from PySide6.QtWidgets import QApplication, QMainWindow, QTextBrowser

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


class WebApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.browser = QTextBrowser(self)
        self.setCentralWidget(self.browser)

        self.setWindowTitle("PySide Web App")
        self.setGeometry(100, 100, 800, 600)


def run_flask():
    app.run(debug=False, threaded=False)


if __name__ == '__main__':
    import threading

    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    app = QApplication(sys.argv)
    window = WebApp()
    window.browser.setOpenExternalLinks(True)
    window.browser.setOpenLinks(False)
    window.browser.setSource('http://127.0.0.1:5000/')
    window.show()
    sys.exit(app.exec())
