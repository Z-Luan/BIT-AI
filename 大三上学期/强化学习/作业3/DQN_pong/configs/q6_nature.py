class config():
    # env config
    render_train     = False
    render_test      = False
    overwrite_render = True
    record           = False
    train            = True
    high             = 255.

    # output config
    output_path  = "results/q6_nature/"
    model_output = output_path + "model.weights"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"

    # model and training config
    num_episodes_test = 20
    grad_clip         = True
    clip_val          = 10
    saving_freq       = 5000
    log_freq          = 50
    eval_freq         = 100
    soft_epsilon      = 0

    # hyper params
    nsteps_train       = 1000
    batch_size         = 32
    buffer_size        = 500
    target_update_freq = 500
    gamma              = 0.99
    learning_freq      = 4
    state_history      = 4
    lr_begin           = 0.00025
    lr_end             = 0.0001
    lr_nsteps          = nsteps_train/2
    eps_begin          = 1
    eps_end            = 0.01
    eps_nsteps         = nsteps_train/2
    learning_start     = 200
    