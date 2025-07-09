import pandas as pd

# ---------- SETTINGS ----------
csv_file = "Orbital_Delivery_Schedule.csv"  # Replace with your actual CSV filename
# -----------------------------

# Load the CSV
df = pd.read_csv(csv_file)

# Create a pivot-style breakdown: rows = dsp_name, columns = product_status, values = counts
breakdown = df.pivot_table(
    index="dsp_name",
    columns="product_status",
    aggfunc="size",
    fill_value=0
)

# Display result
print("✅ DSP-wise Product Status Breakdown:\n")
print(breakdown)

# Optionally, export to CSV
# breakdown.to_csv("dsp_product_status_breakdown.csv")
