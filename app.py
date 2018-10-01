from flask import Flask, render_template, flash, request, jsonify
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import StringType, FloatType

from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel


# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


class ReusableForm(Form):
    name = TextField('Paste Document Contents here:', validators=[validators.required()])


@app.route("/", methods=['GET', 'POST'])
def hello():
    form = ReusableForm(request.form)

    print(form.errors)
    if request.method == 'POST':
        name = request.form['name']
        print(name)

        if form.validate():
            # Save the comment here.
            sc = SparkContext()
            sc.setLogLevel("ERROR")

            app = Flask(__name__)

            schema = StructType([
                StructField("_c0", StringType()),
                StructField("_c1", StringType())
            ])
            predict_schema = StructType([StructField("_c1", StringType())])

            pipelineModel = PipelineModel.load("pipeline_Model")
            lfModel = LogisticRegressionModel.load("lr_Model")

            spark = SparkSession.builder.getOrCreate()
            input_features = [[(name)]]

            predict_df = spark.createDataFrame(data=input_features,
                                               schema=predict_schema)
            transformed_pred_df = pipelineModel.transform(predict_df)
            predictions = lfModel.transform(transformed_pred_df)
            probs = predictions.select('probability').take(1)[0][0]

            n_predictions = len(probs)
            labels = pipelineModel.stages[-1].labels
            result_dict = {labels[i]: probs[i] for i in range(n_predictions)}
            #results = jsonify(result_dict)
            flash(result_dict)

        else:
            flash('All the form fields are required. ')

    return render_template('hello.html', form=form)


if __name__ == "__main__":
    app.run()