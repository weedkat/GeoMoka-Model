from geomoka.inference.engine import inference_evaluate

def test(model, testloader, cfg, class_dict, logger=None):
    model.eval()
    eval_results = inference_evaluate(
        model=model,
        dataloader=testloader,
        ignore_index=cfg.get('ignore_index', 255),
        mode=cfg.get('eval_mode', 'resize'),
        class_dict=class_dict,
        device='cuda',
        verbose=True,
        logger=logger,
        transform_cfg=cfg.get('inference', [])
    )

    return eval_results