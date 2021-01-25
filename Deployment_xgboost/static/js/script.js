var feature_button = d3.select("#buttons").selectAll("button")
var form = d3.select("form")

var yesno_features = ["no_fee","has_roofdeck","has_washer_dryer","has_elevator","has_dishwasher","has_patio","has_gym","has_doorman"]
var yesno = ["Yes","No"]

feature_button.on("click", function() {
    var selection = d3.select(this);
    form.append("label")
        .text(selection.text()+":");
    if(yesno_features.includes(selection.attr("name"))) {
        form.append("select")
            .attr("name",selection.attr("name"))
            .attr("class",selection.attr("name"))
            .append("option")
                .attr("selected","true")
                .attr("disabled","disabled")
                .text("-- select Yes/No --");
        for (i=0;i<2;i++) {
            d3.select(`.${selection.attr("name")}`)
                .append("option")
                .attr("value",i)
                .text(yesno[i]);}
        var select = d3.selectAll("select").select(`".${selection.attr("name")}"`)
        for (i=0;i<2;i++) {
            select.append("option")
                .attr("value",i)
                .text(yesno[i])
        }
        form.append("br");
    } else {
        form.append("input")
            .attr("name",selection.attr("name"))
            .attr("placeholder","--enter value--");
        form.append("br");
    }
})