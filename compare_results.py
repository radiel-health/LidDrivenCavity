import json

# Load both training histories
with open('Models/baseline_4walls/training_history.json') as f:
    baseline = json.load(f)
    
with open('Models/training_history.json') as f:
    new = json.load(f)

print("="*80)
print("📊 TRAINING RESULTS COMPARISON: 4-Wall vs 3-Wall Model")
print("="*80)

print("\n🔵 Baseline (4 walls):")
print(f"  Best val loss:    {min(baseline['val_loss']):.6f} (epoch {baseline['val_loss'].index(min(baseline['val_loss']))+1})")
print(f"  Final val loss:   {baseline['val_loss'][-1]:.6f}")
print(f"  Best train loss:  {min(baseline['train_loss']):.6f}")
print(f"  Epochs trained:   {len(baseline['val_loss'])}")

print("\n🟢 New (3 walls only):")
print(f"  Best val loss:    {min(new['val_loss']):.6f} (epoch {new['val_loss'].index(min(new['val_loss']))+1})")
print(f"  Final val loss:   {new['val_loss'][-1]:.6f}")
print(f"  Best train loss:  {min(new['train_loss']):.6f}")
print(f"  Epochs trained:   {len(new['val_loss'])}")

# Calculate improvement
improvement = (min(baseline['val_loss']) - min(new['val_loss'])) / min(baseline['val_loss']) * 100

print("\n" + "="*80)
if improvement > 0:
    print(f"🎯 IMPROVEMENT: {improvement:.2f}% better (validation loss reduced)")
    print(f"   {min(baseline['val_loss']):.6f} → {min(new['val_loss']):.6f}")
elif improvement < -5:
    print(f"⚠️  REGRESSION: {abs(improvement):.2f}% worse (validation loss increased)")
    print(f"   {min(baseline['val_loss']):.6f} → {min(new['val_loss']):.6f}")
else:
    print(f"➡️  SIMILAR: {abs(improvement):.2f}% difference (roughly equivalent)")
    print(f"   {min(baseline['val_loss']):.6f} → {min(new['val_loss']):.6f}")

print("="*80)

# Training stability
baseline_std = sum([(x - min(baseline['val_loss']))**2 for x in baseline['val_loss'][-20:]]) / 20
new_std = sum([(x - min(new['val_loss']))**2 for x in new['val_loss'][-20:]]) / 20

print("\n📈 Training Stability (last 20 epochs variance):")
print(f"  Baseline: {baseline_std:.6f}")
print(f"  New:      {new_std:.6f}")
if new_std < baseline_std:
    print(f"  ✓ New model is more stable!")
else:
    print(f"  ⚠ Baseline was more stable")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("Removing the top wall from training:")
if improvement > 5:
    print("✅ SIGNIFICANTLY IMPROVED validation loss")
elif improvement > 0:
    print("✓ Slightly improved validation loss")
elif improvement > -5:
    print("➡️ No significant change in validation loss")
else:
    print("❌ Worsened validation loss")
    
print("\nNext steps:")
print("1. Run evaluation on test set: python evaluate.py")
print("2. Compare per-wall metrics (bottom/left/right only)")
print("3. Run inference: python infer.py -re 1000 -ar 1x1 -output results/ -plot")
print("="*80)
