# Laboratory Reports

This directory contains laboratory test reports organized by year, month, and version.

## Directory Structure

```
reports/
├── README.md                    # This file
├── templates/                   # Report templates
│   ├── experiment_report.md
│   └── test_report.md
└── YYYY/                        # Year (e.g., 2025)
    └── MM/                      # Month (01-12, e.g., 11 for November)
        └── vX.Y/                # Version (e.g., v1.0, v1.1)
            ├── report.md        # Main report
            ├── metrics.json     # Test metrics
            ├── plots/           # Charts and graphs
            └── data/            # Raw test data
```

## Current Reports

### 2025

#### November (11)
- **v1.0** - Initial testing and baseline metrics
- **v1.1** - Ensemble model improvements

## Report Naming Convention

- **Year**: 4-digit year (YYYY)
- **Month**: 2-digit month (MM), 01-12
- **Version**: Semantic versioning (vX.Y)
  - Major version (X): Significant changes or new model architecture
  - Minor version (Y): Improvements, optimizations, or bug fixes

## Report Template

Use the template in `templates/experiment_report.md` for new reports.

## Adding a New Report

1. Create directory: `reports/YYYY/MM/vX.Y/`
2. Copy template: `cp reports/templates/experiment_report.md reports/YYYY/MM/vX.Y/report.md`
3. Fill in the report details
4. Add metrics and plots to the version directory
5. Update this README with the new report entry

## Report Contents

Each report should include:

- **Executive Summary**: Key findings and results
- **Test Configuration**: Model parameters, data sources, test environment
- **Metrics**: Accuracy (R²), MAE, RMSE, MAPE, inference time
- **Results**: Comparison with baseline, improvements achieved
- **Charts**: Visualizations of predictions vs actuals, feature importance
- **Conclusions**: Insights and recommendations for next steps

---

**Last Updated**: November 2025

