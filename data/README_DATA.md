# Dataset setup

The full **SMS Spam Collection** dataset is included as `SMSSpamCollection.csv` in this folder. It was converted from the original tab-separated `SMSSpamCollection` file in the ZIP you supplied.

- Rows imported: 5574
- Labels: `ham` (legitimate) and `spam`
- CSV columns: `label`, `message`

Original dataset source: UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

Dataset citation:
Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, T. (2011). Contributions to the Study of SMS Spam Filtering: New Collection and Results. Proceedings of the 2011 ACM Symposium on Document Engineering. DOI: 10.1145/2034691.2034742.

Please retain this citation in your report and check the dataset source page for its terms before redistributing the dataset.

## Cleaning applied by the code
The raw file contains 414 exact-duplicate messages. `load_data()` removes them (5,574 → 5,160 rows: 4,518 ham, 642 spam) so identical texts cannot appear in both the training and test sets.
