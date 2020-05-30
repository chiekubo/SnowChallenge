from torch.utils.data import DataLoader
from dataloaders import datasets


def make_data_loader(args, **kwargs):

	if args.dataset == 'harmo_patch':
		train_set = datasets.harmo_patch(split='train') #(19688)
		val_set   = datasets.harmo_patch(split='val')

		train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs) #(308,)
		val_loader   = DataLoader(  val_set, batch_size=args.batch_size, shuffle=False, **kwargs)

		return train_loader, val_loader

	elif args.dataset == 'harmo_test':
		test_set = datasets.harmo_test()
		test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
		return test_loader


	elif args.dataset == 'harmo_detect':
		train_loader = None
		val_set = datasets.harmo_image(args)
		val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)

		return train_loader, val_loader

	else:
		raise NotImplementedError # for class inheritance
