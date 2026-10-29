"""Presentation formatting; never round measurements before aggregation."""
import csv


class ReportWriter(csv.DictWriter):
    """Render numeric report cells with two decimals, preserving blanks and flags."""

    def writerow(self, rowdict):
        return super().writerow({
            key: f"{value:.2f}" if type(value) in (int, float) else value
            for key, value in rowdict.items()
        })

    def writerows(self, rowdicts):
        for row in rowdicts:
            self.writerow(row)
