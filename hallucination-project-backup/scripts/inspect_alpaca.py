from datasets import load_dataset

print("Loading Alpaca-GPT4...")

dataset = load_dataset("vicgalle/alpaca-gpt4")

print("\nDataset:")
print(dataset)

train = dataset["train"]

print("\nNumber of training examples:")
print(len(train))

print("\nColumns:")
print(train.column_names)

print("\nFirst 5 examples:")
for i in range(5):
    print("\n" + "=" * 70)
    print(f"EXAMPLE {i+1}")
    print("=" * 70)

    example = train[i]

    for key, value in example.items():
        print(f"\n{key}:")
        print(value)
