def missing_data_analysis(df):
    try:
        missing_counts = df.isnull().sum()
        total = len(df)

        missing_data = [
            {
                "column": col,
                "missing_count": int(count),
                "missing_percentage": round((count / total) * 100, 2),
            }
            for col, count in missing_counts.items() if count > 0
        ]

        return {"missing_data": missing_data or "No missing data found"}

    except Exception as e:
        return {"error": f"missing_data_analysis error: {str(e)}"}
