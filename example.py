import tri


print('Testing Question Class...')
question=tri.Question()
print('Default parameters:',question.alpha, question.beta, question.guessing_param)
question.set_alpha(0.5)
question.set_beta(-0.2)
question.set_guessing_param(0.15)
print('Default parameters:',question.alpha, question.beta, question.guessing_param)
print('P(X=1|theta=0):',question.probability_correct(0))
print('Correctness a priori probability:',question.prior_probability_correct())




