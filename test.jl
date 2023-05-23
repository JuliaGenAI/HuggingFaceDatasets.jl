using Test
tmnist = load_dataset("mnist", split="test").with_format("julia")
@test size(tmnist[1]["image"]) == (28, 28)

