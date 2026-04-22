service PredictionService {

  @readonly
  entity ScoredResults {
    key record_id          : Integer;
        order_year         : Integer;
        order_month        : Integer;
        category_name      : String(100);
        shipping_mode      : String(100);
        market             : String(100);
        order_region       : String(100);
        department         : String(100);
        customer_segment   : String(100);
        payment_type       : String(50);
        predicted_demand   : Decimal(15,2);
        category_avg       : Decimal(15,2);
        inventory_decision : String(50);
        delay_risk         : Decimal(8,3);
        delay_decision     : String(50);
        scored_at          : Timestamp;
  };

}