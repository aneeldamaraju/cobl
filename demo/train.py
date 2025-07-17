from cobl import initialize_training

trainer = initialize_training("cobl/model_v5/config.yaml", override_eager=False)
batch = next(iter(trainer.train_dataloader))
trainer.plot_drawn_samples(batch, "test.png")
trainer.fit()
